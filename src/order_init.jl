enlarge_coupling(model, s::Symbol, i::Int, j::Int, bondratio) = enlarge_coupling(model, Val(s), i::Int, j::Int, bondratio)

function enlarge_coupling(model::J1J2, ::Val{:none}, i, j, bondratio)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2, ::Val{:plaquette}, i, j, bondratio)
    J1 = model.J1
    J1h = J1v = J1
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
        J1v = J1 * bondratio 
    elseif (i,j) == (1,2)
        J1v = J1 * bondratio 
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2, ::Val{:dimmer1}, i, j, bondratio)
    J1 = model.J1
    J1h = J1v = J1 
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2, ::Val{:dimmer2}, i, j, bondratio)
    J1 = model.J1
    J1h = J1v = J1 
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
    elseif (i,j) == (2,2)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end


function enlarge_coupling(model::J1J2, ::Val{:mixed}, i, j, bondratio)
    J1 = model.J1
    J1h = J1v = J1
    if (i,j) == (1,1)
        J1h = J1 * bondratio[1]
        J1v = J1 * bondratio[2]
    elseif (i,j) == (1,2)
        J1v = J1 * bondratio[2]
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio[1]
    end

    return J1h, J1v
end
